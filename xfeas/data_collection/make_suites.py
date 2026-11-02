#!/usr/bin/env python
"""Writes the lists of scenes to collect (suites) and the job list for the queue.

Details:
gen5b suites + jobs. Situations are the gen5 training situations (xfeas/_support/data_collection/make_suite5.py: real assets,
supported, training lanes only, test scenes >= 3 m, night >= 60 %); gen5b only re-collects them with the branch schedule.
Extra guard: every school lane is re-checked against the school-course test SC2A (experiments/3_school_course/v2a spec, the
whole course path incl. the first lintel), >= 3 m in plan and |floor dz| < 2 m -> dropped.

Initial batch:
  10  low-obstacle (crawl v2, overhang, env_overhang) x AMO (g1c + variants) and Spot (spotp + variants), schedules A + B
  20  null (C) for the same controllers
  30  GR00T (g1g) low-obstacle A + B
Deferred (NOT queued; append later with --groups): sidestep families, other controllers, Jackal / Iris / terrain / stairs.
    python3 xfeas/data_collection/make_suites.py --groups a,b,c [--per-job 3] [--write-jobs]
Suites: data/gen_v5b/suites/<env>_<base>_<group>.json ; jobs appended to data/gen_v5b/queue/jobs.jsonl (ids unique, resumable).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, json, math, os, sys
from collections import Counter
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
SU5 = os.path.join(ROOT, "data", "gen_v5", "suites")
OUT = os.path.join(ROOT, "data", "gen_v5b", "suites")
Q = os.path.join(ROOT, "data", "gen_v5b", "queue")
SC2A = os.path.join(os.path.dirname(ROOT), "experiments", "3_school_course", "v2a", "school_course_v2a_spec.json")
LOW = ("crawl", "overhang", "env_overhang")
ENVS = ("school", "office", "warehouse_shelves", "full_warehouse", "rivermark")
GROUPS = dict(
    a=dict(prio=10, bases=("g1c", "spotp"), fams=LOW),
    b=dict(prio=20, bases=("g1c", "spotp"), fams=("null",)),
    c=dict(prio=30, bases=("g1g",), fams=LOW),
)
# initial batch sizing (smoke 10-27: AMO ~18 min / pair, Spot ~4 min / pair (A + B), GR00T ~9 min / pair, null 1.5-6 min)
PER_JOB = dict(a=dict(g1c=2, spotp=8), b=dict(g1c=4, spotp=12), c=dict(g1g=3))
MAX_PER_ENV = dict(a=dict(g1c=11, spotp=48), b=dict(g1c=99, spotp=99), c=dict(g1g=9))


def sc2a_poly():
    S = json.load(open(SC2A))["build"]["task"]
    return np.array([p[:2] for p in S["path"]], float), float(S["start"]["z"])


def seg_seg(p0, p1, Q_):
    best = 1e9
    for t in np.linspace(0, 1, 41):
        q = p0 + t * (p1 - p0)
        for i in range(len(Q_) - 1):
            a, b = Q_[i], Q_[i + 1]
            d = b - a
            u = np.clip(((q - a) @ d) / max(d @ d, 1e-9), 0, 1)
            best = min(best, float(np.linalg.norm(a + u * d - q)))
    return best


def load_pool(env, base, fams):
    """low-obstacle groups: the gen5b env_overhang suites (make_env_suites.py, <env>_<base>_b1low.json); null: the gen5 p1
    suites. School is dropped entirely: every school training lane is within 3 m of SC2A (checked below, kept as a guard)."""
    out = []
    if fams == ("null",):
        f = os.path.join(SU5, f"{env}_{base}_p1.json")
        src = [s for s in json.load(open(f)) if s["family5"] == "null"] if os.path.exists(f) else []
    else:
        f = os.path.join(OUT, f"{env}_{base}_b1low.json")
        src = [s for s in json.load(open(f)) if s["family5"] in fams] if os.path.exists(f) else []
    for s in src:
        out.append(dict(s, src_suite=os.path.basename(f)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="a,b,c")
    ap.add_argument("--per-job", type=int, default=0, help="0 = PER_JOB table")
    ap.add_argument("--write-jobs", action="store_true")
    ap.add_argument("--tag", default="b1")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(Q, exist_ok=True)
    poly, zc = sc2a_poly()
    jobs = []
    seen = set()
    for g in a.groups.split(","):
        G = GROUPS[g]
        for base in G["bases"]:
            for env in ENVS:
                S = load_pool(env, base, G["fams"])
                keep, why = [], Counter()
                for s in S:
                    if s["sid"] in seen:
                        why["dup"] += 1
                        continue
                    L = s["lane"]
                    if env == "school":
                        p0 = np.array(L["start"], float)
                        y = float(L["yaw"])
                        p1 = p0 + float(L.get("len", 8.0)) * np.array([math.cos(y), math.sin(y)])
                        if abs(float(L["floor"]) - zc) < 2.0 and seg_seg(p0, p1, poly) < 3.0:
                            why["near_SC2A"] += 1
                            continue
                    if len(keep) >= MAX_PER_ENV[g][base]:
                        why["batch_cap"] += 1
                        continue
                    seen.add(s["sid"])
                    keep.append(s)
                if not keep:
                    continue
                fn = os.path.join(OUT, f"{env}_{base}_{a.tag}{g}.json")
                json.dump(keep, open(fn, "w"))
                n_pairs = sum(len(s["ctrls"]) for s in keep)
                night = np.mean([str(s["lighting"]["mode"]).startswith("night") for s in keep])
                print(
                    f"[suite5b] {g} {env:17s} {base:6s} sits {len(keep):3d} pairs {n_pairs:3d} night {night:.2f} fams {dict(Counter(s['family5'] for s in keep))} dropped {dict(why)}"
                )
                per = PER_JOB[g][base] if a.per_job <= 0 else a.per_job
                for i0 in range(0, len(keep), per):
                    jid = f"{env}_{base}_{a.tag}{g}_{i0:04d}"
                    jobs.append(
                        dict(
                            id=jid,
                            env=env,
                            base=base,
                            suite=os.path.relpath(fn, ROOT),
                            chunk=f"{i0}:{min(i0 + per, len(keep))}",
                            launcher="base",
                            timeout_s=4 * 3600,
                            hang_s=2400,
                            prio=G["prio"] + (0 if base != "g1c" else -1),
                        )
                    )
    print(f"[suite5b] {len(jobs)} jobs")
    if a.write_jobs:
        fj = os.path.join(Q, "jobs.jsonl")
        have = set()
        if os.path.exists(fj):
            have = {json.loads(l)["id"] for l in open(fj) if l.strip()}
        new = [j for j in jobs if j["id"] not in have]
        with open(fj, "a") as f:
            for j in new:
                f.write(json.dumps(j) + "\n")
        print(f"[suite5b] appended {len(new)} jobs -> {fj}")


if __name__ == "__main__":
    main()
