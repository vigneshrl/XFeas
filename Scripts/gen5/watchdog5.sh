#!/bin/bash
# gen_v5 watchdog (server-side, independent of any Claude session; copy of scripts/gen4/watchdog_v4.sh semantics):
# - queue process dead (and work left) -> kill the orphaned jobs' PROCESS GROUPS by the pids recorded in state.json (own user
#   only; never pkill -f), restart the queue (jobs RESUME)
# - audit FAILs never pause the queue: queue5 quarantines them (data/gen_v5/queue/QUARANTINE.txt); the watchdog reports them
# - status line every 10 min to data/gen_v5/queue/watchdog.log; exits when no job is pending / running
cd /p/cral/vignesh/ARL/Scripts
P=/p/cral/vignesh/envs/gsam2/bin/python
Q=data/gen_v5/queue
W=$Q/watchdog.log
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $W; }
counts() { $P -c "import json,collections;st=json.load(open('$Q/state.json'));c=collections.Counter(s['status'] for s in st.values());print(' '.join(f'{k}={v}' for k,v in sorted(c.items())))" 2>/dev/null; }
left() { $P -c "import json;st=json.load(open('$Q/state.json'));print(sum(s['status'] in ('pending','running') for s in st.values()))" 2>/dev/null || echo 1; }
orphans() { $P -c "import json;st=json.load(open('$Q/state.json'));print(' '.join(str(s['pid']) for s in st.values() if s.get('status')=='running' and s.get('pid')))" 2>/dev/null; }
echo $$ > $Q/watchdog.pid
log "watchdog start pid $$ $(counts)"
i=0; nq=0
while true; do
  sleep 120; i=$((i+1))
  n=$(wc -l < $Q/QUARANTINE.txt 2>/dev/null || echo 0)
  if [ "$n" != "$nq" ]; then log "quarantined jobs: $n (queue keeps running): $(tail -1 $Q/QUARANTINE.txt)"; nq=$n; fi
  qpid=$(cat $Q/queue.pid 2>/dev/null)
  if ! kill -0 "$qpid" 2>/dev/null; then
    if [ "$(left)" -gt 0 ]; then
      log "queue pid $qpid DEAD -> killing orphan job process groups + restarting"
      for p in $(orphans); do
        if [ "$(ps -o user= -p $p 2>/dev/null | tr -d ' ')" = "atk9sb" ]; then kill -TERM -- -$p 2>/dev/null; fi
      done
      sleep 20
      for p in $(orphans); do
        if [ "$(ps -o user= -p $p 2>/dev/null | tr -d ' ')" = "atk9sb" ]; then kill -KILL -- -$p 2>/dev/null; fi
      done
      setsid nohup $P scripts/gen5/queue5.py >> $Q/queue.out 2>&1 < /dev/null &
      sleep 10; log "queue restarted pid $(cat $Q/queue.pid)"
    fi
  fi
  [ $((i % 5)) -eq 0 ] && log "status $(counts) threads=$(ps -u atk9sb -L | wc -l) gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
  if [ "$(left)" -eq 0 ]; then log "ALL DONE $(counts)"; exit 0; fi
done
