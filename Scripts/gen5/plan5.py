#!/usr/bin/env python
"""
gen_v5 production PLAN: suites per (env, base) + queue jobs (data/gen_v5/queue/jobs.jsonl). Nothing is launched here.
Target (GEN5_DESIGN 10a): G1 : Spot = 50 : 30 of the final total (Jackal 10 / drone 10 are added on Day 2) -> of the
G1 + Spot episodes 62.5 % G1, 37.5 % Spot. Default 3,200 situations (+ twins 20 %): G1 2,000 = g1c 450, g1g 450, g1l 400,
g1u 350, g1p 350; Spot 1,200. Environment split per base: school 40 %, office 15 %, warehouse_shelves 10 %, full_warehouse 10 %,
rivermark 5 %, terrain 20 %. Chunk = 8 situations per job. Timeouts from data/gen_v5/throughput.json (smoke walls).
    python3 scripts/gen5/plan5.py [--scale 1.0] [--dry]
"""
import argparse, json, math, os, subprocess, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--chunk", type=int, default=8)
ap.add_argument("--dry", action="store_true", help="print the plan only")
a = ap.parse_args()
N_BASE = dict(g1c=450, g1g=450, g1l=400, g1u=350, g1p=200, spotp=1200)   # g1p capped 2026-10-25 (smoke: 7/7 climb falls)
ENV_SPLIT = dict(school=0.40, office=0.15, warehouse_shelves=0.10, full_warehouse=0.10, rivermark=0.05, terrain=0.20)
D5 = os.path.join(ROOT, "data", "gen_v5")
TP = os.path.join(D5, "throughput.json")
WALL = dict(g1c=260, g1g=200, g1l=160, g1u=160, g1p=220, spotp=180)          # s per episode (smoke-measured values override)
if os.path.exists(TP):
    WALL.update({k: float(v) for k, v in json.load(open(TP)).get("wall_s_per_episode_p75", {}).items()})
plan, tot_eps, tot_s = [], 0, 0.0
for b, nb in N_BASE.items():
    for env, f in ENV_SPLIT.items():
        n = int(round(nb * f * a.scale))
        if n <= 0:
            continue
        plan.append((env, b, n)); tot_eps += n * 1.2; tot_s += n * 1.2 * WALL[b]
starts = math.ceil(sum(math.ceil(n / a.chunk) for _, _, n in plan)) * 150
print(f"[plan5] {len(plan)} suites, ~{tot_eps:.0f} episodes (twins incl.), ~{(tot_s + starts) / 3600:.0f} process-hours "
      f"-> ~{(tot_s + starts) / 3600 / 6:.0f} h with 6 concurrent Isaac processes (3 per GPU), ~{(tot_s + starts) / 3600 / 5:.0f} h with 5")
for env, b, n in plan:
    print(f"   {env:18s} {b:6s} {n:4d} situations")
if a.dry:
    sys.exit(0)
Q = os.path.join(D5, "queue"); os.makedirs(Q, exist_ok=True)
have = set()
if os.path.exists(os.path.join(Q, "jobs.jsonl")):
    have = {json.loads(l)["id"] for l in open(os.path.join(Q, "jobs.jsonl")) if l.strip()}
add = 0
with open(os.path.join(Q, "jobs.jsonl"), "a") as fo:
    for env, b, n in plan:
        r = subprocess.run([sys.executable, os.path.join(THIS, "make_suite5.py"), "--env", env, "--base", b, "--n", str(n), "--tag", "p1"],
                           capture_output=True, text=True, cwd=ROOT)
        print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-400:])
        if r.returncode != 0:
            raise SystemExit(f"suite failed {env} {b}")
        suite = f"data/gen_v5/suites/{env}_{b}_p1.json"
        S = json.load(open(os.path.join(ROOT, suite)))
        w = N_BASE[b] / 450.0
        for k, i in enumerate(range(0, len(S), a.chunk)):
            jid = f"{env}_{b}_p1_{i:04d}"
            if jid in have:
                continue
            runs = sum(len(x["ctrls"]) for x in S[i:i + a.chunk])
            fo.write(json.dumps(dict(id=jid, env=env, base=b, suite=suite, chunk=f"{i}:{min(i + a.chunk, len(S))}", launcher="base",
                                     timeout_s=int(1500 + 2.5 * WALL[b] * runs), hang_s=int(max(1800, 4 * WALL[b])),
                                     prio=round(k / w, 3))) + "\n")
            add += 1
print(f"[plan5] added {add} jobs -> data/gen_v5/queue/jobs.jsonl")
