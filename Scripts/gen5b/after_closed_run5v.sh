#!/bin/bash
# 10-27 (user): run the clean redo (run5v.sh, resumes from last.pt) once the SONIC crawl test (pgid 1269164) and the
# gen5b closed-loop queue (queue.log "ALL DONE") have finished. Fallback: start anyway after 4 h.
cd /p/cral/vignesh/ARL/Scripts
L=data/gen_train/gen5b/runs/supervisor_v.log; Q=data/gen5_closed/queue5b/queue.log
echo $$ > data/gen_train/gen5b/runs/after_closed.pid
echo "$(date '+%m-%d %T') waiting for SONIC test + closed-loop queue" >> $L
for i in $(seq 1 480); do
  sonic=$(ps -g 1269164 -o pid= 2>/dev/null | wc -l)
  [ "$sonic" -eq 0 ] && grep -q "ALL DONE" $Q && break
  sleep 30
done
echo "$(date '+%m-%d %T') closed-loop runs finished (or 4 h fallback) -> clean redo" >> $L
exec bash scripts/gen5b/run5v.sh >> $L 2>&1
