#!/bin/bash
# gen_v5 production start -- DO NOT RUN before the user has checked the smoke (analysis/GEN5_DATA_SPEC.md 9).
# Gate: data/probes_v5/MANIFEST.json present (every training controller probed: probe_v4 + transitions).
# 1) suites + jobs (plan5.py), 2) first batch = 2 slots (one school job per robot first: g1c + spotp, prio -1),
# 3) GO + queue + watchdog under setsid/nohup. After the first-batch audits PASS and their contact sheets were looked at:
#      echo 6 > data/gen_v5/queue/max_mine
cd /p/cral/vignesh/ARL/Scripts
P=/p/cral/vignesh/envs/gsam2/bin/python
[ -f data/probes_v5/MANIFEST.json ] || { echo "no data/probes_v5/MANIFEST.json"; exit 1; }
python3 scripts/gen5/plan5.py || exit 1
$P - <<'EOF'
import json
fn = "data/gen_v5/queue/jobs.jsonl"
J = [json.loads(l) for l in open(fn) if l.strip()]
for j in J:
    if j["id"] in ("school_g1c_p1_0000", "school_spotp_p1_0000"):
        j["prio"] = -1
    elif j["base"] == "g1c" and j["prio"] >= 0:          # AMO is the long pole
        j["prio"] = round(j["prio"] / 2, 3)
open(fn, "w").write("".join(json.dumps(j) + "\n" for j in J))
print(len(J), "jobs")
EOF
echo 2 > data/gen_v5/queue/max_mine
touch data/gen_v5/queue/GO
setsid nohup $P scripts/gen5/queue5.py > data/gen_v5/queue/queue.out 2>&1 < /dev/null &
sleep 5
setsid nohup bash scripts/gen5/watchdog5.sh > /dev/null 2>&1 < /dev/null &
echo "queue pid $(cat data/gen_v5/queue/queue.pid) watchdog started"
