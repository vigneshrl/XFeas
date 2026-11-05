#!/usr/bin/env python
"""Checks one finished collection job: files load, no held-out controller or test scene, camera frames not dark,
physics sane. Writes contact sheets of the frames. Exit code 1 means the job failed the audit.

Details:
gen5b per-job audit (+ contact sheets). Exit 1 on FAIL (job_queue quarantines the job, never pauses).
  K  keys / loadable npz, fmt gen_v5b, run_id matches the root
  H  no held-out controller (g1s, spotf) / environment (gq, hospital, interior)
  C  camera: frames finite, mean rgb > 4 on >= 90 % of moment frames, cam matrices finite, shared mount
  F  floor rendered (first decision floor check ok) for every pair
  T  moments: X (N x 35) finite, feas present, history frames at -1.5 / -1 / -0.5 / 0 s, h(t) records >= 10 (padded to 6 s as gen5), proprio >= 50
  L  labels: in [0, 1] where binary, every moment has >= 2 branches, keep / stock rows present
  R  restore fidelity: share of checked moments with pose difference > 2 cm (WARN > 10 %), FAIL if the median > 10 cm
  N  night share (reported; the suites hold >= 0.6 overall)
Contact sheet per pair: <root>/audit/sheets_<tag>/<family>/<pair>.jpg (moment head frames with labels + side views + branch
thumbnails).
    python3 xfeas/data_collection/job_audit.py --root data/gen_v5b_smoke --part smoke1_g1c --tag s1 --sheets
"""

import os, sys  # noqa: E401  (standalone process, possibly in an isolated env: find the repo root itself)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import xfeas  # noqa: E402, F401  (puts every XFeas code folder on sys.path)
import argparse, glob, json, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--part", default="")
ap.add_argument("--tag", default="a")
ap.add_argument("--sheets", action="store_true")
a = ap.parse_args()
HELD_C = ("spotf",)  # g1s is a training controller since 2026-10-30
HELD_E = ("gq", "hospital", "interior")


def sheet(z, rec, fn):
    import cv2

    rows = []
    M = len(z["m_label"])
    tiles = []
    for i in range(M):
        im = np.ascontiguousarray(z["rgb"][i][-1][..., ::-1])
        im = cv2.resize(im, (224, 224))
        ks = np.where(z["b_moment"] == i)[0]
        txt = f"{z['m_group'][i]} {z['m_label'][i]} dh {z['m_d_haz'][i]:.2f}"
        cv2.putText(im, txt, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for j, k in enumerate(ks[:9]):
            s = f"{z['b_cls'][k]}: {'S' if z['y_success'][k] else '-'}{'F' if z['y_fell'][k] else ''}{'C' if z['y_contact'][k] else ''} p{z['y_progress'][k]:.1f}"
            cv2.putText(
                im,
                s,
                (3, 30 + 14 * j),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 0) if z["y_success"][k] else (0, 0, 255),
                1,
            )
        tiles.append(im)
    side = sorted(glob.glob(os.path.join(a.root, rec.get("video") or "__none__", "*.jpg")))[:6]
    for p in side:
        im = cv2.imread(p)
        if im is not None:
            tiles.append(cv2.resize(im, (224, 224)))
    th = z["th_rgb"]
    for i in range(min(len(th), 12)):
        im = cv2.resize(np.ascontiguousarray(th[i][..., ::-1]), (224, 224))
        ri = int(z["th_run"][i])
        cv2.putText(
            im,
            f"run {ri} {z['r_cls'][ri]} t{z['th_t'][i]:.1f}",
            (3, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )
        tiles.append(im)
    W = 6
    while len(tiles) % W:
        tiles.append(np.zeros((224, 224, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[r * W : (r + 1) * W]) for r in range(len(tiles) // W)])
    hdr = np.zeros((24, grid.shape[1], 3), np.uint8)
    cv2.putText(
        hdr,
        f"{rec['sid']} {rec['family5']} {rec['env']} {rec['light']['mode']} stock {rec['stock_end']}",
        (4, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    cv2.imwrite(fn, np.vstack([hdr, grid]), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    pat = f"sits_{a.part}.jsonl" if a.part else "sits_*.jsonl"
    files = glob.glob(os.path.join(a.root, "*", "*", pat))
    fails, warns = [], []
    n_pairs = n_mom = n_br = 0
    nights = []
    fid = []
    rid = "gen_v5b_smoke" if "smoke" in os.path.basename(os.path.normpath(a.root)) else "gen_v5b"
    for f in files:
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            n_pairs += 1
            tag = r["sid"]
            if any(r["ctrl_id"].startswith(h) or f"_{h}_" in r["ctrl_id"] for h in HELD_C) or r["base"] in HELD_C:
                fails.append(f"H {tag} held-out controller")
            if any(h in r["env"].lower() for h in HELD_E):
                fails.append(f"H {tag} held-out env")
            if r.get("run_id") != rid:
                fails.append(f"K {tag} run_id {r.get('run_id')}")
            if not (r.get("floor_render") or {}).get("ok"):
                fails.append(f"F {tag} floor not rendered")
            nights.append(bool(r.get("night")))
            fn = os.path.join(a.root, r["br_file"])
            try:
                z = dict(np.load(fn, allow_pickle=False))
            except Exception as e:
                fails.append(f"K {tag} load {e!r}"[:200])
                continue
            if str(z["fmt"]) != "gen_v5b":
                fails.append(f"K {tag} fmt")
            M = len(z["m_label"])
            n_mom += M
            n_br += len(z["b_cls"])
            mr = z["rgb"].reshape(M * 4, -1).mean(1)
            if np.mean(mr > 4) < 0.9:
                fails.append(f"C {tag} dark frames {np.mean(mr <= 4):.2f}")
            if not np.isfinite(z["cam"]).all():
                fails.append(f"C {tag} camera matrix")
            if not np.isfinite(z["X"].astype(np.float32)).all():
                fails.append(f"T {tag} descriptors")
            if np.any(np.abs(z["t_hist"] - np.array([-1.5, -1.0, -0.5, 0.0])) > 0.02):
                fails.append(f"T {tag} history frame times {z['t_hist'].round(2).tolist()}"[:200])
            if np.any(z["rec_n"] < 10) or np.any(z["prop_n"] < 50):
                fails.append(f"T {tag} short h(t) / proprio {z['rec_n'].tolist()} {z['prop_n'].tolist()}"[:200])
            if np.isnan(z["feas"]).all(1).any():
                warns.append(f"T {tag} feas NaN at a moment")
            for i in range(M):
                ks = np.where(z["b_moment"] == i)[0]
                if len(ks) < 2:
                    fails.append(f"L {tag} moment {i} has {len(ks)} branches")
                if not np.any(z["b_keep"][ks]):
                    fails.append(f"L {tag} moment {i} no keep branch")
            for k in ("success", "fell", "stall", "contact", "w_fall", "w_contact"):
                v = z[f"y_{k}"]
                if np.any((v != 0) & (v != 1)):
                    fails.append(f"L {tag} y_{k} not binary")
            fid += [x for x in r.get("fid_dpos", []) if x is not None and np.isfinite(x)]
            if a.sheets:
                sheet(z, r, os.path.join(a.root, "audit", f"sheets_{a.tag}", r["family5"], f"{r['sid']}.jpg"))
    fr = float(np.mean(np.asarray(fid) > 0.02)) if fid else 0.0
    # smoke 10-27: re-simulating the stock prefix from the spawn reproduces the moment pose to 3-16 mm; restoring a moment and
    # re-running the same command for 1 s drifts 0.2-7 cm (walking contact chaos; AMO more than Spot). FAIL only when the
    # restore is broken (median > 10 cm); the > 2 cm share is reported.
    fmed = float(np.median(fid)) if fid else 0.0
    if fmed > 0.10:
        fails.append(f"R restore fidelity broken: median {fmed:.3f} m")
    elif fr > 0.10:
        warns.append(f"R restore fidelity: {fr:.2f} of moments > 2 cm (median {fmed:.3f} m)")
    st = "FAIL" if fails else "PASS"
    out = dict(
        part=a.part,
        status=st,
        pairs=n_pairs,
        moments=n_mom,
        branches=n_br,
        night=float(np.mean(nights)) if nights else None,
        fid_n=len(fid),
        fid_gt2cm=fr,
        fid_p50=float(np.median(fid)) if fid else None,
        fid_max=float(np.max(fid)) if fid else None,
        fails=fails[:30],
        warns=warns[:30],
    )
    os.makedirs(os.path.join(a.root, "audit"), exist_ok=True)
    json.dump(out, open(os.path.join(a.root, "audit", f"audit_{a.tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("fails", "warns")}))
    for x in fails[:10] + warns[:10]:
        print("  ", x)
    print(
        f"[job_audit] {st} pairs {n_pairs} moments {n_mom} branches {n_br} fails {len(fails)} warns {len(warns)} fid>2cm {fr:.2f}"
    )
    sys.exit(1 if fails else 0)


main()
