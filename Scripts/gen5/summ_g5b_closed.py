#!/usr/bin/env python
"""Summary of the gen5b closed-loop matrix (run_g5b_closed.sh): per cell (robot x arm x scene) success / falls / TTG from
runs_<robot>.jsonl and, from diag/<jid>/decisions5_t<k>.jsonl, the decision behaviour:
  first crouch   d_to_overhang (lintel front - base x) at the first EXECUTED command whose pelvis_height is below nominal
  switches       number of executed command changes per run (flip-flop check)
  stand-up       first return to an upright command after a crouch: base x relative to the lintel END (+ = past it)
  crouched frac  fraction of decisions with a crouch command executed (O1N: over-intervention)
    python scripts/gen5/summ_g5b_closed.py [--prefix c5_] [--json out.json]
"""
import argparse, glob, json, os
import numpy as np

G5 = "/p/cral/vignesh/ARL/Scripts/data/gen5_closed"
NOM_H = dict(g1a=0.75, spotp=0.52)


def trial_stats(dec, rob):
    hn = NOM_H[rob]
    low = lambda r: float(r["executed"].get("pelvis_height", hn)) < hn - 1e-6
    first = next((r for r in dec if low(r)), None)
    sw, prev = 0, None
    for r in dec:
        k = json.dumps(r["executed"], sort_keys=True)
        if prev is not None and k != prev:
            sw += 1
        prev = k
    up = None
    if first is not None:
        i0 = dec.index(first)
        up = next((r for r in dec[i0:] if not low(r)), None)
    return dict(first_crouch_d=None if first is None else first["d_to_overhang"],
                switches=sw, standup_past_end=None if up is None else up["past_overhang_end"],
                crouched_frac=round(float(np.mean([low(r) for r in dec])), 2) if dec else None, n_dec=len(dec),
                cls_seq=_runs([r["chosen"]["cls"] for r in dec]))


def _runs(seq):
    out = []
    for s in seq:
        if out and out[-1][0] == s:
            out[-1][1] += 1
        else:
            out.append([s, 1])
    return " ".join(f"{s}x{n}" for s, n in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="c5_")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = []
    for d in sorted(glob.glob(os.path.join(G5, "course", a.prefix + "*"))):
        if not os.path.isdir(d) or "_try" in os.path.basename(d):
            continue
        jid = os.path.basename(d)
        for f in glob.glob(os.path.join(d, "runs_*.jsonl")):
            rob = os.path.basename(f)[5:-6]
            R = [json.loads(l) for l in open(f) if l.strip()]
            T = []
            for r in R:
                k = int(r["trial"])
                p = os.path.join(G5, "diag", jid, f"decisions5_t{k}.jsonl")
                dec = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
                T.append(dict(trial=k, outcome=r["outcome"], success=bool(r["success"]), t_end=r["t_end"], dist_to_goal=r.get("dist_to_goal"),
                              **trial_stats(dec, rob)))
            ok = [t for t in T if t["success"]]
            res.append(dict(jid=jid, robot=rob, n=len(T), success=sum(t["success"] for t in T), falls=sum(t["outcome"] == "fall" for t in T),
                            ttg=None if not ok else round(float(np.mean([t["t_end"] for t in ok])), 1), trials=T))
    for c in res:
        print(f"{c['jid']:22s} {c['robot']:6s} SR {c['success']}/{c['n']} falls {c['falls']} TTG {c['ttg']}")
        for t in c["trials"]:
            print(f"    t{t['trial']} {t['outcome']:8s} t_end {t['t_end']:6.1f} crouch@d {t['first_crouch_d']} switches {t['switches']} "
                  f"stand-up past end {t['standup_past_end']} crouched {t['crouched_frac']} | {t['cls_seq'][:160]}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
