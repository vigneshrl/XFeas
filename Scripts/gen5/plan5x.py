#!/usr/bin/env python
"""
gen_v5 production PLAN for the wheeled / aerial bases (2026-10-26; plan5.py is not modified). GEN5_DESIGN 10a: G1 50 /
Spot 30 / Jackal 10 / drone 10 -> Jackal = drone = 1/3 of Spot's 1,200 situations = 400 situations each (+ twins 20 %).
Suites via make_suite5x.py (tag p1), jobs appended to data/gen_v5/queue/jobs.jsonl in plan5's format (8 situations per
job), prio 1000 + k so that every G1 / Spot job (prio < 1000) is started first. The running queue reloads jobs.jsonl.
    python3 scripts/gen5/plan5x.py [--dry]
"""
import argparse, json, os, subprocess, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--dry", action="store_true")
ap.add_argument("--prio0", type=float, default=1000.0)
a = ap.parse_args()
N_BASE = dict(jackal=400, iris=400)
ENV_SPLIT = dict(jackal=dict(school=0.30, office=0.15, warehouse_shelves=0.10, full_warehouse=0.10, rivermark=0.10, terrain=0.25),
                 iris=dict(school=0.45, office=0.20, warehouse_shelves=0.15, full_warehouse=0.20))
D5 = os.path.join(ROOT, "data", "gen_v5")
WALL = dict(jackal=60.0, iris=90.0)
TP = os.path.join(D5, "throughput.json")
if os.path.exists(TP):
    WALL.update({k: float(v) for k, v in json.load(open(TP)).get("wall_s_per_episode_p75", {}).items() if k in WALL})
plan = []
for b, nb in N_BASE.items():
    for env, f in ENV_SPLIT[b].items():
        plan.append((env, b, int(round(nb * f))))
tot_s = sum(n * 1.2 * WALL[b] for _, b, n in plan) + sum(-(-n // a.chunk) for _, _, n in plan) * 150
print(f"[plan5x] {len(plan)} suites, ~{sum(n for *_, n in plan) * 1.2:.0f} episodes, ~{tot_s / 3600:.1f} process-hours "
      f"(~{tot_s / 3600 / 6:.1f} h on 6 slots)")
for env, b, n in plan:
    print(f"   {env:18s} {b:6s} {n:4d} situations")
if a.dry:
    sys.exit(0)
Q = os.path.join(D5, "queue")
have = {json.loads(l)["id"] for l in open(os.path.join(Q, "jobs.jsonl")) if l.strip()}
rows = []
k = 0
for env, b, n in plan:
    r = subprocess.run([sys.executable, os.path.join(THIS, "make_suite5x.py"), "--env", env, "--base", b, "--n", str(n), "--tag", "p1"],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-600:])
    if r.returncode != 0:
        raise SystemExit(f"suite failed {env} {b}")
    suite = f"data/gen_v5/suites/{env}_{b}_p1.json"
    S = json.load(open(os.path.join(ROOT, suite)))
    for i in range(0, len(S), a.chunk):
        jid = f"{env}_{b}_p1_{i:04d}"
        if jid in have:
            continue
        runs = sum(len(x["ctrls"]) for x in S[i:i + a.chunk])
        rows.append(dict(id=jid, env=env, base=b, suite=suite, chunk=f"{i}:{min(i + a.chunk, len(S))}", launcher="base",
                         timeout_s=int(1500 + 2.5 * WALL[b] * runs), hang_s=int(max(1800, 4 * WALL[b])), prio=None, _i=i))
# interleave the two bases and the environments so both robots progress together; prio 1000 + k
rows.sort(key=lambda r: (r["_i"], r["base"], r["env"]))
for k, r in enumerate(rows):
    r["prio"] = round(a.prio0 + k, 1); r.pop("_i")
with open(os.path.join(Q, "jobs.jsonl"), "a") as fo:
    fo.write("".join(json.dumps(r) + "\n" for r in rows))
print(f"[plan5x] added {len(rows)} jobs (prio {a.prio0:.0f}..{a.prio0 + len(rows) - 1:.0f}) -> data/gen_v5/queue/jobs.jsonl")
