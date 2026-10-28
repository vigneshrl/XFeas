#!/usr/bin/env python
"""
gen5 DATA LOADER (tick-level). analysis/GEN5_DATA_SPEC.md 7.

One ITEM = one decision tick of one gen_v5 episode:
  context    the gen4 context dict, built by scripts/gen4/dataset_gen4.build_context (UNCHANGED) from a gen4-format obs dict
             that is assembled from the episode arrays at that tick (obs_at): current head frame (224), the 3 history frames
             at -1.5 / -1.0 / -0.5 s (rgb resized to 128 as gen_v2 rgb_hist, min-pooled depth128), per-frame camera
             matrices / base poses / eye heights, 2 s proprio (100 x 50 Hz), h(t) (60 x 10 Hz, VRob records packed with
             virtual.pack_records), feas v0.4 (16), spec / probe (probes_v5 office_v4.json: fingerprint, ops, trace)
  candidates cand (N, F_DESC) = the logged descriptors of the tick (describe5.DESC_KEYS), cand_mask, chosen (index)
  targets    labels5 (y_fall, y_stall, y_contact, y_contact_Ns, y_progress, y_success, outcome, term_in_h, next_dec) for the
             CHOSEN candidate; next_row = the row index of the tick 1.5 s later in the same manifest (-1 if terminal or
             not in the manifest) for bootstrapped value targets
Splits: zone_split (train / val lanes) inside each base; leave-one-base-out folds over BASES_TRAIN, every v_ / lk_ / mm_
variant goes with its base. SONIC (g1s) and the Isaac Spot flat policy (spotf) can never enter (build() raises).

    PY=/p/cral/vignesh/envs/gsam2/bin/python
    $PY scripts/gen5/dataset5.py build --root data/gen_v5 --out data/gen_train/gen5/manifest.jsonl [--smoke]
    $PY scripts/gen5/dataset5.py check --manifest data/gen_train/gen5/manifest.jsonl --n 16
"""
import argparse, glob, json, math, os, sys
from collections import defaultdict, OrderedDict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
for _p in (THIS, os.path.join(ROOT, "scripts", "gen4"), os.path.join(ROOT, "scripts", "gen_train"), os.path.join(ROOT, "scripts", "gen3"),
           os.path.join(ROOT, "scripts", "gen"), os.path.join(ROOT, "scripts", "xembody")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ctrl5 as K5
import labels5 as LB
import describe5 as DS
import virtual as VC
import dataset_gen4 as DG
import gen4_common as G4C

BASES_TRAIN = K5.BASES
HELD = K5.HELD_OUT
HELD_ENV = ("gq", "hospital", "interior")
K_PROP = 100

import canon5                                                # gen5 canonical joints (G1 + Spot + Jackal wheels + Iris rotors)
XBODY_DESC = {b: os.path.join(THIS, f"{b}_desc.json") for b in K5.XBODY_BASES}
_XDESC = {}


def robot_desc5(robot):
    """fixed embodiment descriptor per robot: G1 / Spot from dataset_gen4, Jackal / Iris from scripts/gen5/<robot>_desc.json"""
    if robot in XBODY_DESC:
        if robot not in _XDESC:
            _XDESC[robot] = json.load(open(XBODY_DESC[robot]))
        return _XDESC[robot]
    return DG.robot_desc(robot)


def crawl_v1(r, part):
    """crawl situations of the p1 suite (fixed underside band, mostly impassable by construction; GEN5_DATA_SPEC v0.3); the v2
    generator's jobs are the *_crawl2_* parts. Keyed on the part, not pol_commit: p1 crawl jobs run after the fix carry pol_commit too"""
    return r.get("family5") == "crawl" and "_crawl2_" not in part


def _robot(base):
    return "spot" if base.startswith("spot") else "g1"


# ------------------------------------------------------------------------------------------------ manifest
def finished_parts(root, smoke=False):
    if smoke:
        return None, {}
    p = os.path.join(root, "queue", "state.json")
    st = json.load(open(p)) if os.path.exists(p) else {}
    ok = [k for k, v in st.items() if v.get("status") == "done"]
    return set(ok), {k: v.get("status") for k, v in st.items() if v.get("status") != "done"}


def build(root, smoke=False, exclude=None, every=1, keep_crawl_v1=False, tilt_fall=1.2):        # tilt_fall 1.2: user 2026-10-26
    root = os.path.abspath(root)
    parts, why = finished_parts(root, smoke)
    rows, drop = [], defaultdict(int)
    excl = set(exclude or [])
    seen = set()                       # a sid in two parts shares one npz (crawl2 reused 5 p1 crawl sids and overwrote their files)
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "sits_*.jsonl")), key=lambda f: ("_crawl2_" not in f, f)):   # crawl2 first: it owns shared sids
        part = os.path.basename(f)[5:-6]
        if parts is not None and part not in parts:
            drop["part_not_done"] += 1; continue
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if K5.is_held_out(r["ctrl_id"]) or r["base"] in HELD:
                raise SystemExit(f"[gen5 manifest] {r['sid']}: held-out controller {r['ctrl_id']} in {root} -- never used for training")
            if any(h in r["env"].lower() for h in HELD_ENV):
                raise SystemExit(f"[gen5 manifest] {r['sid']}: held-out environment {r['env']}")
            if r.get("run_id") != ("gen_v5_smoke" if smoke else "gen_v5"):
                drop["run_id"] += 1; continue
            if r["sid"] in excl or r["geo_sit"] in excl:
                drop["qa_excluded"] += 1; continue
            if not keep_crawl_v1 and crawl_v1(r, part):
                drop["crawl_v1"] += 1; continue
            if r["sid"] in seen:
                drop["dup_sid"] += 1; continue
            seen.add(r["sid"])
            ep = os.path.join(root, r["ep_file"])
            if not os.path.exists(ep):
                drop["ep_missing"] += 1; continue
            z = np.load(ep, allow_pickle=False)
            zd = {k: z[k] for k in ("t", "dec", "path_world", "t_clock0", "rec_t", "rec_ach", "arc", "end_reason", "t_end", "bin_s",
                                    "bins_haz_body", "bins_haz_foot", "chosen", "hazards_json", "bins_scene_body", "tilt", "robot")}
            Lb = LB.labels(zd, tilt_fall=tilt_fall)
            spec = K5.spec_path(r["ctrl_id"])
            probe = os.path.join(ROOT, r["probe_file"])
            assert os.path.abspath(probe).startswith(K5.PROBES_V5 + os.sep), probe
            key0 = len(rows)
            ticks = list(Lb["tick_index"])
            pos = {int(k): i for i, k in enumerate(ticks)}
            for i, k in enumerate(ticks):
                if every > 1 and i % every:
                    continue
                y = {kk: (float(Lb[kk][i]) if np.isfinite(Lb[kk][i]) else None) for kk in ("y_fall", "y_stall", "y_contact", "y_contact_Ns",
                                                                                         "y_contact_foot", "y_progress", "term_in_h", "t_to_end", "y_success")}
                rows.append(dict(key=f"{r['sid']}#{int(k)}", ep=ep, tick=int(k), sid=r["sid"], ctrl_id=r["ctrl_id"], base=r["base"], robot=r["robot"],
                                 virtual=r["virtual"], env=r["env"], family5=r["family5"], aggr=r.get("aggr", False), split=r.get("zone_split", "train"),
                                 light=r["light"]["mode"], spec=spec, probe=probe, trans=os.path.join(ROOT, r["trans_file"]), chosen=int(Lb["chosen"][i]),
                                 outcome=int(Lb["outcome"][i]), end=Lb["end_reason"], next_tick=int(Lb["next_dec"][i]), **y))
            # next_row links (inside this episode's rows)
            km = {rows[j]["tick"]: j for j in range(key0, len(rows))}
            for j in range(key0, len(rows)):
                rows[j]["next_row"] = km.get(rows[j]["next_tick"], -1)
    return rows, dict(drop), why


def fold_rows(rows, holdout_base=""):
    out = []
    for r in rows:
        r = dict(r)
        if holdout_base and K5.base_of(r["ctrl_id"]) == holdout_base:
            r["split"] = "test"
        out.append(r)
    return out


def set_weights(rows):
    n = defaultdict(int)
    for r in rows:
        n[(r["ctrl_id"], r["family5"])] += 1
    for r in rows:
        r["weight"] = 1.0 / math.sqrt(n[(r["ctrl_id"], r["family5"])])
    return rows


# ------------------------------------------------------------------------------------------------ obs at a tick
class EpCache:
    def __init__(self, n=4):
        self.n = n; self.D = OrderedDict()

    def get(self, p):
        if p in self.D:
            self.D.move_to_end(p); return self.D[p]
        z = np.load(p, allow_pickle=False)
        z = {k: z[k] for k in z.files}
        self.D[p] = z
        if len(self.D) > self.n:
            self.D.popitem(last=False)
        return z


def obs_at(z, k):
    """gen4-format obs dict (run_gen_v4.write_obs keys used by dataset_gen4.build_context) of tick k of episode z"""
    import cv2
    import clearmap as CMg
    hist = [max(0, k + int(o)) for o in np.asarray(z["hist_ticks"])]
    t = np.asarray(z["t"], np.float64)
    tk = t[k]
    o = dict(rgb=z["rgb"][k], depth=np.asarray(z["depth"][k], np.float32), cam_dec=np.asarray(z["cam"][k], np.float64))
    o["t_hist"] = np.array([t[i] - tk for i in hist], np.float32)
    o["rgb_hist"] = np.stack([cv2.resize(np.asarray(z["rgb"][i]), (128, 128), interpolation=cv2.INTER_AREA) for i in hist]).astype(np.uint8)
    o["depth_hist"] = np.stack([z["depth128"][i] for i in hist]).astype(np.float16)
    o["cam_hist"] = np.stack([z["cam"][i] for i in hist]).astype(np.float64)
    o["base_hist_world"] = np.stack([z["base_world"][i] for i in hist]).astype(np.float64)
    o["cam_h_hist"] = np.array([z["cam_h"][i] for i in hist], np.float32)
    o["selfmask_hist"] = np.array([z["selfmask"][i] for i in hist], np.float32)
    o["K128"] = CMg.intrinsics(128); o["K224"] = CMg.intrinsics(224)
    pn = int(z["prop_n"][k])
    o["prop_base"] = np.asarray(z["prop_base"][max(0, pn - K_PROP):pn], np.float32)
    o["prop_joints"] = np.asarray(z["prop_joints"][max(0, pn - K_PROP):pn], np.float32)
    hn = int(z["hist_n"][k])
    R = [(float(z["rec_t"][i]), z["rec_cmd"][i], z["rec_env"][i], z["rec_ach"][i], z["rec_q"][i] if i < len(z["rec_q"]) else np.zeros(0),
          z["rec_dq"][i] if i < len(z["rec_dq"]) else np.zeros(0)) for i in range(min(hn, len(z["rec_t"])))]
    geo = dict(P0=np.asarray(z["P0"]), d=np.asarray(z["d"]), n=np.asarray(z["n"]), yaw=float(z["lane_yaw"]))
    H = VC.pack_records(R, geo, float(z["floor_z"]), float(z["t_clock0"]) + tk, VC.HIST_N)
    for kk, v in H.items():
        o[f"hist_{kk}"] = v
    o["joint_names"] = z["joint_names"]; o["ach_keys"] = z["ach_keys"]; o["hist_roles"] = z["hist_roles"]
    o["feas"] = np.asarray(z["feas"][k], np.float32)
    o["cam_source"] = z["cam_source"]; o["cam_mount"] = z["cam_mount"]
    return o


T_SWITCH_CAP = 5.0


def decision_state(z, k):
    """(2,) float32: time since the last command switch before tick k (/ T_SWITCH_CAP, capped at 1), any switch yet"""
    t = np.asarray(z["t"], np.float64); sw = np.asarray(z["switched"], bool)[:k]
    j = np.where(sw)[0]
    dt = (t[k] - t[j[-1]]) if len(j) else t[k]
    return np.array([min(max(dt, 0.0), T_SWITCH_CAP) / T_SWITCH_CAP, float(len(j) > 0)], np.float32)


def compute_norm(rows, n=200, seed=0):
    rng = np.random.default_rng(seed)
    tr = [r for r in rows if r["split"] == "train"] or rows
    pick = [tr[i] for i in rng.choice(len(tr), min(n, len(tr)), replace=False)] if tr else []
    B, Jc = [], []
    cache = EpCache()
    for r in pick:
        z = cache.get(r["ep"])
        o = obs_at(z, r["tick"])
        desc = robot_desc5(r["robot"])
        pb = np.asarray(o["prop_base"], np.float32); pj = np.asarray(o["prop_joints"], np.float32)
        if len(pb) == 0:
            continue
        jc, jm = G4C.canon_joints(DG.joint_names_of(o, desc, pj.shape[1]), pj, axis=1)
        B.append(pb); Jc.append(jc[:, jm].reshape(-1, pj.shape[-1]))
    st = lambda x: (x.mean(0).tolist(), np.maximum(x.std(0), 1e-3).tolist())
    nb, nj = (st(np.concatenate(B)), st(np.concatenate(Jc))) if B else (([0.0] * 13, [1.0] * 13), ([0.0] * 7, [1.0] * 7))
    fps = {}
    for r in tr:
        if r["ctrl_id"] not in fps:
            fps[r["ctrl_id"]] = G4C.fp_vector(json.load(open(r["probe"])))
    if fps:
        F = np.stack([v[0] for v in fps.values()]); Pm = np.stack([v[1] for v in fps.values()])
        mu = np.array([F[Pm[:, i], i].mean() if Pm[:, i].any() else 0.0 for i in range(G4C.N_FP)])
        sd = np.array([max(F[Pm[:, i], i].std(), 0.02) if Pm[:, i].sum() > 1 else 1.0 for i in range(G4C.N_FP)])
    else:
        mu, sd = np.zeros(G4C.N_FP), np.ones(G4C.N_FP)
    return dict(base_mean=nb[0], base_std=nb[1], joint_mean=nj[0], joint_std=nj[1], fp_mean=mu.tolist(), fp_std=sd.tolist(),
                n_fp_ctrls=len(fps), fp_ctrls=sorted(fps))


Y_KEYS = ("y_fall", "y_stall", "y_contact", "y_contact_Ns", "y_contact_foot", "y_progress", "term_in_h", "t_to_end", "y_success")

try:
    import torch
    from torch.utils.data import Dataset

    class TickDataset(Dataset):
        def __init__(self, rows, norm, cfg=None, augment=False, aug=None, seed=0):
            self.rows, self.norm, self.cfg, self.augment = rows, norm, dict(cfg or {}), augment
            self.aug = dict(DG.DEFAULT_AUG, **(aug or {}))
            self._rng = np.random.default_rng(seed)
            self._spec, self._probe, self._ptr = {}, {}, {}
            self.cache = EpCache()

        def __len__(self):
            return len(self.rows)

        def _json(self, d, p):
            if p not in d:
                d[p] = json.load(open(p))
            return d[p]

        def __getitem__(self, i):
            r = self.rows[i]
            assert not K5.is_held_out(r["ctrl_id"])
            z = self.cache.get(r["ep"])
            k = r["tick"]
            o = obs_at(z, k)
            spec = self._json(self._spec, r["spec"])
            probe = self._json(self._probe, r["probe"]) if self.cfg.get("use_fp", True) else None
            desc = robot_desc5(r["robot"])
            ctx, feas_raw = DG.build_context(o, spec, probe, desc, self.norm, self.cfg, train=self.augment, rng=self._rng, aug=self.aug,
                                             ptrace_cache=self._ptr)
            X = np.asarray(z["X"][k], np.float32).copy()
            # is_keep / is_nominal are positional (index 0 / 1); smoke data before the describe5 cache-key fix (10-23
            # 01:10) could carry them on other rows with the same command -> normalised here
            X[:, DS.KI["is_keep"]] = 0.0; X[0, DS.KI["is_keep"]] = 1.0
            X[:, DS.KI["is_nominal"]] = 0.0; X[1, DS.KI["is_nominal"]] = 1.0
            t = torch.from_numpy
            out = {kk: t(np.asarray(v)) for kk, v in ctx.items()}
            out.update(cand=t(X), chosen=torch.tensor(int(z["chosen"][k]), dtype=torch.long), idx=i, state=t(decision_state(z, k)),
                       cur_cmd=t(np.nan_to_num(np.asarray(z["cmd"][k], np.float32), nan=0.0)),
                       next_row=torch.tensor(int(r.get("next_row", -1)), dtype=torch.long),
                       outcome=torch.tensor(int(r["outcome"]), dtype=torch.long),
                       feas_raw=t(feas_raw.astype(np.float32) if feas_raw is not None else np.full(G4C.N_FEAS, np.nan, np.float32)))
            for kk in Y_KEYS:
                v = r.get(kk)
                out[kk] = torch.tensor(float("nan") if v is None else float(v), dtype=torch.float32)
            return out

    def collate5(batch):
        B = len(batch)
        K = max(b["rgb"].shape[0] for b in batch)
        N = max(b["cand"].shape[0] for b in batch)
        out = {}
        rgb = torch.zeros(B, K, 3, 224, 224, dtype=torch.uint8); dep = torch.zeros(B, K, 1, 224, 224)
        fmask = torch.zeros(B, K, dtype=torch.bool); fage = torch.zeros(B, K)
        for i, b in enumerate(batch):
            k = b["rgb"].shape[0]
            rgb[i, :k] = b["rgb"]; dep[i, :k] = b["depth"]; fmask[i, :k] = True; fage[i, :k] = b["fage"]
        out.update(rgb=rgb, depth=dep, fmask=fmask, fage=fage)
        for k in ("cmap", "meta", "base", "joints", "jstatic", "jmask", "body", "feas", "spec_tok", "spec_mask", "fp", "fp_mask", "ops",
                  "ops_mask", "ptrace", "ptrace_valid", "hist", "hist_valid", "feas_raw", "cur_cmd", "state"):
            out[k] = torch.stack([b[k] for b in batch])
        cand = torch.zeros(B, N, DS.F_DESC); cm = torch.zeros(B, N, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b["cand"].shape[0]; cand[i, :n] = b["cand"]; cm[i, :n] = True
        out.update(cand=cand, cand_mask=cm)
        for k in ("chosen", "next_row", "outcome") + Y_KEYS:
            out[k] = torch.stack([b[k] for b in batch])
        out["idx"] = torch.tensor([b["idx"] for b in batch])
        return out
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--root", default=os.path.join(ROOT, "data", "gen_v5"))
    b.add_argument("--out", required=True)
    b.add_argument("--smoke", action="store_true")
    b.add_argument("--every", type=int, default=1, help="keep every k-th decision tick (sub-sampling)")
    b.add_argument("--keep-crawl-v1", action="store_true", help="keep the p1 crawl episodes (dropped by default)")
    b.add_argument("--tilt-fall", type=float, default=1.2, help="relabel G1 episodes with tilt above this (rad) as fell; <= 0 turns it off")
    s = sub.add_parser("check")
    s.add_argument("--manifest", required=True)
    s.add_argument("--n", type=int, default=16)
    s.add_argument("--fold", default="", help="leave-one-base-out: holdout base")
    a = ap.parse_args()
    if a.cmd == "build":
        rows, drop, why = build(a.root, smoke=a.smoke, every=a.every, keep_crawl_v1=a.keep_crawl_v1, tilt_fall=a.tilt_fall if a.tilt_fall and a.tilt_fall > 0 else None)
        set_weights(rows)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")
        cnt = defaultdict(lambda: defaultdict(int))
        for r in rows:
            cnt[r["base"]][r["family5"]] += 1
        json.dump(dict(drop=drop, n_rows=len(rows), n_eps=len({r["sid"] for r in rows}), not_done=why), open(a.out + ".info.json", "w"), indent=1)
        print(f"[gen5 manifest] {len(rows)} ticks from {len({r['sid'] for r in rows})} episodes -> {a.out}; dropped {dict(drop)}")
        for k, v in sorted(cnt.items()):
            print(f"   {k:6s} {dict(v)}")
    elif a.cmd == "check":
        rows = [json.loads(l) for l in open(a.manifest)]
        rows = fold_rows(rows, a.fold)
        assert not any(K5.is_held_out(r["ctrl_id"]) for r in rows)
        norm = compute_norm(rows, n=40)
        rng = np.random.default_rng(0)
        pick = [rows[i] for i in rng.choice(len(rows), min(a.n, len(rows)), replace=False)]
        ds = TickDataset(pick, norm, {}, augment=True)
        items = [ds[i] for i in range(len(ds))]
        bt = collate5(items)
        bad = []
        for k, v in bt.items():
            if hasattr(v, "shape"):
                fin = bool(torch.isfinite(v.float()).all()) if v.dtype != torch.bool else True
                if not fin and not k.startswith("y_"):
                    bad.append(k)
                print(f"{k:14s} {tuple(v.shape)} {v.dtype} finite={fin}")
        # round trips on the loader-assembled obs: (1) feas v0.4 recomputed from obs_at's 4 frames (depth128, camera
        # matrices, per-frame bases) == the logged feas; (2) descriptors recomputed by describe5 from the logged command in
        # force before the decision + the logged speed == the logged X (float16 tolerance); (3) chosen index in range
        os.environ.setdefault("XFEAS_PERC_MINPOOL_FIX", "1")         # the collector's perception flags (run_isaac5.sh;
        os.environ.setdefault("XFEAS_SELF_MASK", "exact")            # camera_mount sets clearmap.EXACT_SELF_MASK in-process)
        import candidates5 as CA
        import feas_mem as FM
        n_ok, d_feas, d_X = 0, [], []
        for r in pick:
            z = ds.cache.get(r["ep"]); k = r["tick"]
            X = np.asarray(z["X"][k], np.float32)
            assert X.shape[1] == DS.F_DESC and 0 <= int(z["chosen"][k]) < X.shape[0]
            o = obs_at(z, k)
            P = json.load(open(r["probe"])); T = json.load(open(r["trans"])); spec = json.load(open(r["spec"]))
            frames = [(o["depth_hist"][i].astype(np.float32), o["cam_hist"][i], tuple(map(float, o["base_hist_world"][i]))) for i in range(4)]
            bw = z["base_world"][k]
            Pm = FM.mem_points(frames, (float(bw[0]), float(bw[1]), float(bw[2])), float(z["floor_z"]), self_x=0.0)
            fv, _, _ = FM.feas_from_points(Pm, P["fp"], P["caps"], rules="v04")
            d_feas.append(float(np.nanmax(np.abs(np.asarray(fv, np.float32) - np.asarray(z["feas"][k], np.float32)))))
            prev = json.loads(str(z["cmd_json"][k - 1])) if k > 0 else CA.nominal_cmd(spec)
            Xr = DS.Describer(spec, P, T).describe(CA.enumerate_candidates(spec), prev, float(z["v_b"][k][0]))
            d_X.append(float(np.max(np.abs(Xr.astype(np.float16).astype(np.float32) - X))))
            n_ok += 1
        print(f"[check] feas recomputed from loader obs: max |diff| {max(d_feas):.4g}; descriptors recomputed: max |diff| {max(d_X):.4g} "
              f"(describe5 changes after collection show up here)")
        print(f"[check] {len(pick)} items, non-finite (non-label) keys: {bad}; chosen index in range for {n_ok}/{len(pick)}; "
              f"labels: fall {np.nanmean([r['y_fall'] for r in pick]):.2f} contact {np.nanmean([r['y_contact'] for r in pick]):.2f} "
              f"next_row ok {np.mean([r['next_row'] >= 0 for r in pick]):.2f}")


if __name__ == "__main__":
    main()
