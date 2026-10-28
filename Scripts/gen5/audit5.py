#!/usr/bin/env python
"""
gen_v5 batch AUDIT + contact sheets (plain python + cv2). analysis/GEN5_DATA_SPEC.md 8. Every record and every episode
npz of the selection is checked (not a sample). Exit 3 = FAIL (the production queue QUARANTINES the job, never pauses).

  python3 scripts/gen5/audit5.py --root data/gen_v5 --part <job id> [--sheets]         (production, per job)
  python3 scripts/gen5/audit5.py --root data/gen_v5_smoke --tag smoke --sheets          (everything under the root)
Checks (FAIL unless noted):
  K  keys of GEN5_DATA_SPEC present, fmt gen_v5, run_id = gen_v5 (gen_v5_smoke under the smoke root)
  H  no held-out controller (g1s, spotf), no held-out environment, probes from data/probes_v5 (office_v4 + transitions)
  C  camera matrix per tick finite with orthonormal rotation rows; mount g1_head_php (G1) / spot_body (Spot) / jackal_front / iris_front; self-mask
     fraction finite and <= 0.45 on every frame
  F  floor rendered at the first decision tick (floor_render.ok); builder checks: no primitive, supported (gen_v4 build
     rejects otherwise; re-checked from geo checks where present)
  T  ticks: t steps 0.25 s, >= 1 decision tick, chosen index in range on every decision tick, descriptors finite, feas finite
     on >= 90 % of decision ticks
  L  labels computable (labels5) and in range; contact bins finite >= 0; end_reason valid
  N  night share >= 60 % per robot (checked once >= 20 episodes of that robot)            (WARN below 20)
  E  headlight exposure: night first-decision frames with > 60 % near-black or > 25 % saturated pixels <= 10 % per env
     (checked with >= 10 night episodes per env)
  S  switching: >= 50 % of episodes with >= 1 command switch                                  (WARN)
Contact sheets (--sheets): per episode one jpg = 8 head-camera frames over the episode (tick time, chosen class, policy
mode, near flag, |vx|) + up to 4 side-view frames; written to <root>/audit/sheets_<tag>/<family5>/<sid>.jpg
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THIS)
import ctrl5 as K5
import labels5 as LB

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.join(K5.ROOT, "data", "gen_v5"))
ap.add_argument("--part", default="")
ap.add_argument("--tag", default="")
ap.add_argument("--sheets", action="store_true")
a = ap.parse_args()
ROOT5 = os.path.abspath(a.root)
SMOKE = "smoke" in os.path.basename(ROOT5)
RUN_ID = "gen_v5_smoke" if SMOKE else "gen_v5"
TAG = a.tag or (f"job_{a.part}" if a.part else "all")
KEYS = ("fmt", "run_id", "sid", "ctrl_id", "base", "robot", "spec_json", "probe_file", "trans_file", "fingerprint", "cands_json", "desc_keys", "t",
        "dec", "arc", "pos", "v_b", "tilt", "chosen", "pol_mode", "prop", "cmd", "feas", "X", "rgb", "depth", "depth128", "cam", "selfmask",
        "cam_h", "base_world", "rec_t", "rec_cmd", "rec_ach", "rec_q", "prop_base", "prop_joints", "path_world", "hazards_json", "arc_goal",
        "end_reason", "t_end", "bins_haz_body", "contact_json", "hist_n", "prop_n", "cam_mount", "light_mode")
ENDS = ("success", "fell", "stall", "timeout", "crossed_low")
FAIL, WARN = defaultdict(list), defaultdict(list)


def fail(code, msg):
    FAIL[code].append(msg)


def pix(rgb):
    m = rgb.astype(np.float32).mean(-1)
    return float((m < 10).mean()), float((rgb >= 250).any(-1).mean())


def sheet(z, rec, out):
    import cv2
    T = len(z["t"]); idx = np.unique(np.linspace(0, T - 1, 8).astype(int))
    tiles = []
    cands = json.loads(str(z["cands_json"]))
    for k in idx:
        im = np.ascontiguousarray(z["rgb"][k][..., ::-1]).copy()
        ch = int(z["chosen"][k])
        lab = str(z["cls_new"][k]) if ch >= 0 else "warm"
        txt = [f"t {float(z['t'][k]):.2f} {str(z['pol_mode'][k])[:6]}", f"{lab[:20]}", f"near {int(z['near'][k])} vx {float(z['v_b'][k][0]):+.2f} d {float(z['d_haz'][k]):.1f}"]
        for i, s in enumerate(txt):
            cv2.putText(im, s, (3, 14 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3); cv2.putText(im, s, (3, 14 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        tiles.append(im)
    while len(tiles) < 8:
        tiles.append(np.zeros_like(tiles[0]))
    row1 = np.concatenate(tiles[:4], 1); row2 = np.concatenate(tiles[4:8], 1)
    rows = [row1, row2]
    if rec.get("video"):
        vd = os.path.join(ROOT5, rec["video"])
        fs = sorted(glob.glob(os.path.join(vd, "*.jpg")))
        if fs:
            pick = [fs[i] for i in np.unique(np.linspace(0, len(fs) - 1, 4).astype(int))]
            ims = [cv2.resize(cv2.imread(f), (224, 126)) for f in pick]
            while len(ims) < 4:
                ims.append(np.zeros_like(ims[0]))
            rows.append(np.concatenate(ims, 1))
    head = np.zeros((26, 896, 3), np.uint8)
    cv2.putText(head, f"{rec['sid'][:70]} {rec['family5']} {rec['light']['mode']} -> {rec['end']} t {rec['t_end']} sw {rec['n_switch']} passed {rec['hazards_passed']}",
                (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.concatenate([head] + rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    files = sorted(glob.glob(os.path.join(ROOT5, "*", "*", f"sits_{a.part}.jsonl" if a.part else "sits_*.jsonl")))
    recs = [json.loads(l) for f in files for l in open(f) if l.strip()]
    rejs = [json.loads(l) for f in glob.glob(os.path.join(ROOT5, "*", "*", f"rejects_{a.part}.jsonl" if a.part else "rejects_*.jsonl")) for l in open(f) if l.strip()]
    by_robot = defaultdict(list); by_env_night = defaultdict(list); n_sw = 0
    fam = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in recs:
        sid = r["sid"]
        if K5.is_held_out(r["ctrl_id"]):
            fail("H", f"{sid}: held-out controller")
        if any(h in r["env"].lower() for h in ("gq", "hospital", "interior")):
            fail("H", f"{sid}: held-out env")
        if r.get("run_id") != RUN_ID:
            fail("K", f"{sid}: run_id {r.get('run_id')}")
        for k in ("probe_file", "trans_file"):
            if not r[k].startswith("data/probes_v5/"):
                fail("H", f"{sid}: {k} {r[k]}")
        fr = r.get("floor_render") or {}
        if not fr.get("ok"):
            fail("F", f"{sid}: floor_render {fr}")
        ch = r.get("checks") or {}
        npc = ch.get("no_primitive") or {}
        if npc and not npc.get("ok", True):
            fail("F", f"{sid}: no_primitive {npc}")
        if ch.get("supports") is not None and not (ch["supports"].get("check") or {}).get("ok", True):
            fail("F", f"{sid}: unsupported")
        p = os.path.join(ROOT5, r["ep_file"])
        if not os.path.exists(p):
            fail("K", f"{sid}: episode file missing"); continue
        z = np.load(p, allow_pickle=False)
        miss = [k for k in KEYS if k not in z.files]
        if miss:
            fail("K", f"{sid}: missing keys {miss}"); continue
        z = {k: z[k] for k in z.files}
        if str(z["fmt"]) != "gen_v5" or str(z["run_id"]) != RUN_ID:
            fail("K", f"{sid}: fmt/run_id {z['fmt']} {z['run_id']}")
        M = np.asarray(z["cam"], np.float64)
        Rm = M[:, :3, :3]
        if not np.isfinite(M).all() or np.abs(np.einsum("nij,nkj->nik", Rm, Rm) - np.eye(3)).max() > 1e-3:
            fail("C", f"{sid}: camera matrices not finite / orthonormal")
        mount = str(z["cam_mount"])
        if mount != dict(g1="g1_head_php", jackal="jackal_front", iris="iris_front").get(r["robot"], "spot_body"):
            fail("C", f"{sid}: mount {mount}")
        sm = np.asarray(z["selfmask"], np.float32)
        if not np.isfinite(sm).all() or sm.max() > 0.45:
            fail("C", f"{sid}: selfmask max {np.nanmax(sm):.3f} finite {np.isfinite(sm).all()}")
        t = np.asarray(z["t"], np.float64); dec = np.asarray(z["dec"], bool)
        if len(t) > 1 and np.abs(np.diff(t) - 0.25).max() > 0.03:
            fail("T", f"{sid}: tick spacing {np.diff(t).min():.3f}..{np.diff(t).max():.3f}")
        if dec.sum() < 1:
            fail("T", f"{sid}: no decision tick (end {z['end_reason']} t {float(z['t_end']):.2f})")
        N = z["X"].shape[1]
        chs = np.asarray(z["chosen"])[dec]
        if len(chs) and (chs.min() < 0 or chs.max() >= N):
            fail("T", f"{sid}: chosen index out of range")
        if not np.isfinite(np.asarray(z["X"], np.float32)[dec]).all():
            fail("T", f"{sid}: descriptors not finite")
        fe = np.asarray(z["feas"], np.float32)[dec]
        if len(fe) and np.isfinite(fe).all(1).mean() < 0.9:
            fail("T", f"{sid}: feas finite on {np.isfinite(fe).all(1).mean():.2f} of decision ticks")
        if str(z["end_reason"]) not in ENDS:
            fail("L", f"{sid}: end_reason {z['end_reason']}")
        b = np.asarray(z["bins_haz_body"])
        if not np.isfinite(b).all() or (b < 0).any():
            fail("L", f"{sid}: contact bins")
        try:
            L = LB.labels(z)
            for k in ("y_fall", "y_contact"):
                v = L[k]
                if not np.all((v == 0) | (v == 1)):
                    fail("L", f"{sid}: {k} not binary")
        except Exception as e:
            fail("L", f"{sid}: labels failed {e!r}")
        by_robot[r["robot"]].append(bool(r["night"]))
        if r["night"] and dec.any():
            k0 = int(np.where(dec)[0][0])
            bl, sa = pix(np.asarray(z["rgb"][k0]))
            by_env_night[r["env"]].append(bl > 0.60 or sa > 0.25)
        n_sw += int(r["n_switch"] > 0)
        f5 = fam[r["family5"]][r["ctrl_id"]]; f5[0] += 1; f5[1] += int(r["success"])
        if a.sheets:
            sheet(z, r, os.path.join(ROOT5, "audit", f"sheets_{TAG}", r["family5"], f"{sid}.jpg"))
    for rb, v in by_robot.items():
        if len(v) >= 20 and np.mean(v) < 0.60:
            fail("N", f"night share {rb} {np.mean(v):.2f} (n {len(v)})")
        elif len(v) < 20:
            WARN["N"].append(f"night share {rb} {np.mean(v):.2f} (n {len(v)} < 20, not checked)")
    for env, v in by_env_night.items():
        if len(v) >= 10 and np.mean(v) > 0.10:
            fail("E", f"exposure {env}: {np.mean(v):.2f} of night first-decision frames black / saturated")
    if recs and n_sw / len(recs) < 0.5:
        WARN["S"].append(f"only {n_sw}/{len(recs)} episodes switch")
    verdict = "PASS" if not FAIL else "FAIL"
    rej = defaultdict(int)
    for x in rejs:
        rej[x["reason"].split(" ")[0][:40]] += 1
    rep = dict(verdict=verdict, tag=TAG, root=os.path.relpath(ROOT5, K5.ROOT), n_episodes=len(recs), n_rejects=len(rejs), rejects=dict(rej),
               fail={k: v[:20] for k, v in FAIL.items()}, warn=dict(WARN), night={k: round(float(np.mean(v)), 3) for k, v in by_robot.items()},
               family_success={f: {c: f"{s}/{n}" for c, (n, s) in d.items()} for f, d in fam.items()})
    os.makedirs(os.path.join(ROOT5, "audit"), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT5, "audit", f"audit_{TAG}.json"), "w"), indent=1, default=str)
    print(f"[audit5] {verdict} {TAG}: {len(recs)} episodes, {len(rejs)} rejects {dict(rej)}; fail {{{', '.join(f'{k}: {len(v)}' for k, v in FAIL.items())}}} "
          f"warn {dict(WARN)}")
    for k, v in FAIL.items():
        for m in v[:5]:
            print(f"   FAIL {k}: {m}")
    sys.exit(0 if verdict == "PASS" else 3)


main()
