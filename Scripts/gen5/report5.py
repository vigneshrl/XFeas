#!/usr/bin/env python
"""
gen5 collection report (plain python): per family5 x controller -> episodes, success, fell, stall, hazard passed (the
robot got past the first hazard's end), clean (success without hazard body contact), and for every PASSED hazard the
command classes in force while the robot was on it (which candidates the controller actually needed); wall seconds per
episode (p50 / p75 per base -> <root>/throughput.json for plan5.py); decision ticks; switch counts.
    python3 scripts/gen5/report5.py --root data/gen_v5_smoke [--md out.md]
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THIS)
import ctrl5 as K5

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.join(K5.ROOT, "data", "gen_v5_smoke"))
ap.add_argument("--md", default="")
a = ap.parse_args()
R = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.root, "*", "*", "sits_*.jsonl"))) for l in open(f) if l.strip()]
J = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.root, "*", "*", "rejects_*.jsonl"))) for l in open(f) if l.strip()]
T = defaultdict(lambda: defaultdict(list))
on_haz = defaultdict(Counter)
for r in R:
    T[r["family5"]][r["ctrl_id"]].append(r)
    if r["hazards_passed"]:
        p = os.path.join(a.root, r["ep_file"])
        try:
            z = np.load(p, allow_pickle=False)
            H = json.loads(str(z["hazards_json"]))
            arc = np.asarray(z["arc"]); cls = np.asarray(z["cls_new"]); dec = np.asarray(z["dec"])
            h = H[0]
            m = (arc >= h["arc0"] - 0.3) & (arc <= h["arc1"] + 0.1) & dec
            for c in cls[m]:
                on_haz[(r["family5"], r["ctrl_id"])][str(c) if c else "nominal"] += 1
        except Exception:
            pass
lines = ["| family5 | controller | episodes | success | fell | stall/timeout | hazard passed | clean | classes in force on the passed hazard (ticks) |",
         "|---|---|---|---|---|---|---|---|---|"]
for f in sorted(T):
    for c in sorted(T[f]):
        L = T[f][c]; n = len(L)
        s = sum(x["success"] for x in L); fl = sum(x["end"] == "fell" for x in L); st = sum(x["end"] in ("stall", "timeout", "crossed_low") for x in L)
        hp = sum(bool(x["hazards_passed"]) or (x["success"] and x["family5"] != "null") for x in L)   # on-top goals end ON the hazard
        cl = sum(bool(x.get("clean")) for x in L)
        oh = ", ".join(f"{k} {v}" for k, v in on_haz[(f, c)].most_common(4))
        lines.append(f"| {f} | {c} | {n} | {s} | {fl} | {st} | {hp} | {cl} | {oh} |")
wall = defaultdict(list)
for r in R:
    wall[r["base"]].append(r["wall_s"])
thr = dict(wall_s_per_episode_p50={b: round(float(np.percentile(v, 50)), 1) for b, v in wall.items()},
           wall_s_per_episode_p75={b: round(float(np.percentile(v, 75)), 1) for b, v in wall.items()},
           realtime_factor={b: round(float(np.median([r["wall_s"] / max(r["t_end"], 1e-3) for r in R if r["base"] == b])), 2) for b in wall},
           n={b: len(v) for b, v in wall.items()}, src=os.path.relpath(a.root, K5.ROOT))
json.dump(thr, open(os.path.join(a.root, "throughput.json"), "w"), indent=1)
rej = Counter(x["reason"].split(" ")[0][:40] for x in J)
summ = [f"episodes {len(R)}, rejects {len(J)} {dict(rej)}",
        f"decision ticks {sum(r['n_dec'] for r in R)}, switches {sum(r['n_switch'] for r in R)}, styles {dict(Counter(r['style'] for r in R))}",
        f"night share {np.mean([r['night'] for r in R]):.2f}" if R else "",
        f"wall s / episode p50 {thr['wall_s_per_episode_p50']} p75 {thr['wall_s_per_episode_p75']} realtime x {thr['realtime_factor']}",
        f"prof (s, mean per episode) " + json.dumps({k: round(float(np.mean([r['prof'][k] for r in R if r.get('prof')])), 1)
                                                     for k in (R[0].get('prof') or {})}) if R and R[0].get("prof") else ""]
out = "\n".join(summ + [""] + lines)
print(out)
if a.md:
    open(a.md, "w").write(out + "\n")
