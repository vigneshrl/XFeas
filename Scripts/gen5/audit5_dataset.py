#!/usr/bin/env python
"""
gen5 DATASET-LEVEL audit (after every job passed audit5). Catches what a per-job audit cannot: frames / episodes copied
across jobs, twins that are identical, frozen physics or camera, broken arrays, label-relevant inconsistencies.

Per episode (multiprocess): md5 of rgb at 3 ticks + depth at the middle tick, base displacement, camera displacement vs base
displacement, NaN / inf in X / feas on decision ticks, chosen in range, t_end vs tick clock, tilt max, night flag vs mean rgb.
Then across episodes: duplicate frame hashes (between different sids), identical twins, per-base summaries.

    python3 scripts/gen5/audit5_dataset.py --root data/gen_v5 --out analysis/gen5_dataset_audit.json [--procs 16]
"""
import argparse, glob, hashlib, json, os
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np


def one(args):
    root, r = args
    p = os.path.join(root, r["ep_file"])
    out = dict(sid=r["sid"], geo=r["geo_sit"], base=r["base"], env=r["env"], fam=r["family5"], end=r["end"], night=r["night"])
    try:
        z = np.load(p, allow_pickle=False)
        t = z["t"]; dec = z["dec"].astype(bool); T = len(t)
        rgb = z["rgb"]
        ks = sorted({0, T // 2, T - 1})
        out["h_rgb"] = [hashlib.md5(np.ascontiguousarray(rgb[k]).tobytes()).hexdigest() for k in ks]
        out["h_dep"] = hashlib.md5(np.ascontiguousarray(z["depth"][T // 2]).tobytes()).hexdigest()
        out["rgb_std_mid"] = float(rgb[T // 2].astype(np.float32).std())
        out["rgb_mean"] = float(rgb[::max(1, T // 8)].astype(np.float32).mean())
        pos = np.asarray(z["pos"], np.float64)
        out["disp"] = float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))
        cam = np.asarray(z["cam"], np.float64)
        ce = cam[:, 3, :3]                                   # USD row-vector convention: translation in the last row
        out["cam_disp"] = float(np.linalg.norm(ce[-1, :2] - ce[0, :2]))
        out["cam_base_gap"] = float(np.max(np.linalg.norm(ce[:, :2] - pos[:, :2], axis=1)))
        X = np.asarray(z["X"], np.float32)[dec]
        out["X_bad"] = int((~np.isfinite(X)).sum())
        fe = np.asarray(z["feas"], np.float32)[dec]
        out["feas_nan_frac"] = float(np.isnan(fe).mean()) if fe.size else 1.0
        ch = np.asarray(z["chosen"])[dec]
        out["chosen_bad"] = int(((ch < 0) | (ch >= X.shape[1])).sum()) if X.size else -1
        out["n_dec"] = int(dec.sum()); out["T"] = T
        out["t_end"] = float(z["t_end"]); out["t_last"] = float(t[-1])
        out["tilt_max"] = float(np.max(z["tilt"]))
        d = np.asarray(z["depth"][T // 2], np.float32)
        out["depth_valid"] = float(np.mean(np.isfinite(d) & (d > 0.05) & (d < 50)))
        out["ok"] = True
    except Exception as e:
        out["ok"] = False; out["err"] = f"{type(e).__name__}: {e}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/gen_v5")
    ap.add_argument("--out", default="analysis/gen5_dataset_audit.json")
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    R = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.root, "*", "*", "sits_*.jsonl"))) for l in open(f) if l.strip()]
    sids = Counter(r["sid"] for r in R)
    with Pool(a.procs) as pool:
        E = pool.map(one, [(a.root, r) for r in R], chunksize=4)
    bad = [e for e in E if not e["ok"]]
    good = [e for e in E if e["ok"]]
    # cross-episode duplicates
    by_h = defaultdict(set)
    for e in good:
        for h in e["h_rgb"][1:]:                             # tick 0 can legitimately repeat for the same spawn (twins / controllers)
            by_h[h].add(e["sid"])
        by_h["d" + e["h_dep"]].add(e["sid"])
    dup = {h: sorted(s) for h, s in by_h.items() if len(s) > 1}
    flags = defaultdict(list)
    for e in good:
        if e["rgb_std_mid"] < 2: flags["blank_frame"].append(e["sid"])
        if e["X_bad"]: flags["X_nonfinite"].append(e["sid"])
        if e["chosen_bad"]: flags["chosen_out_of_range"].append(e["sid"])
        if e["depth_valid"] < 0.5: flags["depth_mostly_invalid"].append(e["sid"])
        if e["cam_base_gap"] > 1.5: flags["camera_off_robot"].append(e["sid"])
        if e["end"] == "success" and e["disp"] < 1.0: flags["success_without_motion"].append(e["sid"])
        if abs(e["cam_disp"] - e["disp"]) > 1.0: flags["camera_not_following"].append(e["sid"])
        if e["t_end"] > e["t_last"] + 0.3: flags["t_end_after_last_tick"].append(e["sid"])
        if e["n_dec"] < 2: flags["too_few_decisions"].append(e["sid"])
    base = defaultdict(Counter)
    for e in good:
        base[e["base"]]["n"] += 1; base[e["base"]][e["end"]] += 1; base[e["base"]]["night"] += bool(e["night"])
    rep = dict(n_rows=len(R), n_unique_sids=len(sids), duplicate_sid_rows={k: v for k, v in sids.items() if v > 1},
               load_errors=[(e["sid"], e["err"]) for e in bad], n_dup_hashes=len(dup), dup_examples=list(dup.items())[:20],
               flags={k: dict(n=len(v), ex=v[:10]) for k, v in flags.items()}, per_base={k: dict(v) for k, v in base.items()})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    json.dump(E, open(a.out.replace(".json", "_episodes.json"), "w"))
    print(json.dumps({k: (v if k != "dup_examples" else v[:5]) for k, v in rep.items() if k != "per_base"}, indent=1)[:6000])
    for k, v in sorted(base.items()):
        print(k, dict(v))


if __name__ == "__main__":
    main()
